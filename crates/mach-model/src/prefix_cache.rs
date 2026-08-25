// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! Paged-prefix KV cache: SHA-256 prefix hashing + prefix-reuse index +
//! full-attention prefix matcher.
//!
//! Ported from LightSeek TokenSpeed ts-scheduler-core (MIT):
//! `tokenspeed-scheduler-rs/crates/ts-scheduler-core/src/prefix_hasher.rs`,
//! `prefix_index.rs` and `prefix_matcher.rs`.
//!
//! The three pieces form the prefix-cache admission path of a paged KV cache:
//!
//! 1. [`hash_prefix_page`] frames a token page — the previous page's digest,
//!    the page tokens and optional extra keys — into one self-delimiting byte
//!    stream and digests it, so every cached page gets a content-addressable
//!    key.
//! 2. [`PrefixCacheIndex`] maps a page's [`CacheKey`] to the canonical
//!    [`CacheBlockLocation`] that holds its KV data (one index per cache
//!    group).
//! 3. [`PrefixMatcher`] walks a request's page keys left-to-right and reports
//!    the longest contiguous, hole-free run of cached pages (full attention).
//!
//! The hasher framing is byte-for-byte compatible with the upstream
//! implementation, so hash keys produced here match hashes produced there.

#![forbid(unsafe_code)]

use crate::kv_block_pool::CacheBlockLocation;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PrefixHasher
// ---------------------------------------------------------------------------

const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";

/// Appends each byte of `bytes` to `out` as two lowercase hex characters.
pub fn append_hex_bytes(out: &mut String, bytes: &[u8]) {
    out.reserve(bytes.len() * 2);
    for b in bytes {
        out.push(HEX_CHARS[(b >> 4) as usize] as char);
        out.push(HEX_CHARS[(b & 0x0f) as usize] as char);
    }
}

/// Decodes a hex string back into raw bytes (inverse of [`digest_to_hex`]).
///
/// Accepts upper- and lowercase hex; an invalid digit panics. An odd trailing
/// nibble is ignored, mirroring the upstream decoder.
pub fn hex_to_bytes(hex: &str) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    let mut iter = hex.bytes();
    while let (Some(hi), Some(lo)) = (iter.next(), iter.next()) {
        let hi = hex_digit(hi);
        let lo = hex_digit(lo);
        bytes.push((hi << 4) | lo);
    }
    bytes
}

fn hex_digit(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        b'A'..=b'F' => c - b'A' + 10,
        _ => panic!("invalid hex digit: {:?}", c as char),
    }
}

/// Encodes a digest as a lowercase hex string.
pub fn digest_to_hex(digest: &[u8]) -> String {
    let mut out = String::with_capacity(digest.len() * 2);
    append_hex_bytes(&mut out, digest);
    out
}

/// Absorbs a `u32` into the hash as four little-endian bytes.
fn sha256_update_u32_le(ctx: &mut Sha256, v: u32) {
    ctx.update([v as u8, (v >> 8) as u8, (v >> 16) as u8, (v >> 24) as u8]);
}

/// Hashes one prefix page into a 64-char lowercase hex digest.
///
/// `prior_hash` is the previous page's digest (empty for the first page);
/// `extra_keys` are optional per-page distinguishing keys (e.g. a LoRA name
/// or cache salt). The framed byte stream is:
///
/// ```text
/// [prior_len u32le][prior bytes][token_count u32le][tokens u32le...]
/// [extra_count u32le][per key: len u32le + key bytes...]
/// ```
///
/// `extra_keys` is the terminal block, so an empty list is skipped without
/// ambiguity (a non-empty list always writes a count >= 1 first).
pub fn hash_prefix_page(tokens: &[i32], prior_hash: &str, extra_keys: &[&str]) -> String {
    let mut ctx = Sha256::new();

    let prior_bytes = hex_to_bytes(prior_hash);
    sha256_update_u32_le(&mut ctx, prior_bytes.len() as u32);
    if !prior_bytes.is_empty() {
        ctx.update(&prior_bytes);
    }

    sha256_update_u32_le(&mut ctx, tokens.len() as u32);
    for token in tokens {
        sha256_update_u32_le(&mut ctx, *token as u32);
    }

    if !extra_keys.is_empty() {
        sha256_update_u32_le(&mut ctx, extra_keys.len() as u32);
        for key in extra_keys {
            sha256_update_u32_le(&mut ctx, key.len() as u32);
            ctx.update(key.as_bytes());
        }
    }

    let digest = ctx.finalize();
    digest_to_hex(&digest)
}

/// Computes the hash chain over `prefix_pages`, seeded by `prior` (the digest
/// of the page before the first one; usually empty).
pub fn compute_prefix_hashes(
    prefix_pages: &[&[i32]],
    prior: &str,
    extra_keys_per_page: &[&[&str]],
) -> Vec<String> {
    let mut hashes = Vec::with_capacity(prefix_pages.len());
    let mut current_prior = prior.to_string();
    for (i, page) in prefix_pages.iter().enumerate() {
        let extra = extra_keys_per_page.get(i).copied().unwrap_or(&[]);
        let hash = hash_prefix_page(page, &current_prior, extra);
        hashes.push(hash.clone());
        current_prior = hash;
    }
    hashes
}

/// Continues an existing hash chain and returns only
/// `prefix_pages[first_page..past_end_page)`'s digests.
///
/// Panics when the range is empty or extends past `prefix_pages`.
pub fn advance_prefix_hashes(
    prefix_pages: &[&[i32]],
    first_page: usize,
    prior: &str,
    past_end_page: usize,
) -> Vec<String> {
    assert!(first_page < past_end_page, "hash range must be non-empty");
    assert!(
        past_end_page <= prefix_pages.len(),
        "hash range exceeds the available full pages"
    );
    compute_prefix_hashes(&prefix_pages[first_page..past_end_page], prior, &[])
}

// ---------------------------------------------------------------------------
// PrefixCacheIndex
// ---------------------------------------------------------------------------

/// Lookup key for one cached prefix page.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Namespace the sequence belongs to (e.g. model instance id).
    pub namespace_id: u32,
    /// Cache group this page belongs to (must match the index's group).
    pub group_id: u32,
    /// SHA-256 prefix digest (64 lowercase hex chars) of the page's token
    /// prefix, as produced by [`hash_prefix_page`].
    pub content_hash: String,
    /// Token offset of the page within the sequence.
    pub page_offset: i32,
}

/// One cached page: its key, the canonical location holding its KV data, and
/// the epoch of its most recent access (used for LRU eviction).
#[derive(Debug, Clone, PartialEq, Eq)]
struct CacheEntry {
    key: CacheKey,
    location: CacheBlockLocation,
    last_access_epoch: u64,
}

/// One cache group's prefix-reuse index: `CacheKey -> canonical CacheBlockLocation`.
///
/// `entries` owns each entry once; `by_key` and `by_location` are non-owning
/// secondary indices (Vec positions). Erase uses `swap_remove` and repairs the
/// moved element's map indices ([`PrefixCacheIndex::remove`]), so all stored
/// indices stay valid. Entry iteration order is not semantically load-bearing:
/// eviction picks the entry with the oldest `last_access_epoch` (LRU), never
/// a Vec position.
#[derive(Debug)]
pub struct PrefixCacheIndex {
    group_id: u32,
    entries: Vec<CacheEntry>,
    by_key: HashMap<CacheKey, usize>,
    by_location: HashMap<CacheBlockLocation, usize>,
    /// Monotonic access counter backing `last_access_epoch`.
    next_epoch: u64,
}

impl PrefixCacheIndex {
    /// Creates an empty index for one cache group.
    #[must_use]
    pub fn new(group_id: u32) -> Self {
        Self {
            group_id,
            entries: Vec::new(),
            by_key: HashMap::new(),
            by_location: HashMap::new(),
            next_epoch: 0,
        }
    }

    /// The cache group this index serves.
    #[must_use]
    pub fn group_id(&self) -> u32 {
        self.group_id
    }

    /// Inserts `(key, location)`.
    ///
    /// Returns the canonical location the caller should keep:
    /// - `None` when `key` was new, or was already canonical at `location`
    ///   (the entry's recency is refreshed);
    /// - `Some(previous)` when `key` already maps to a different canonical
    ///   location, meaning `location` is a duplicate of the same content and
    ///   should be released by the caller.
    ///
    /// Panics when `key` does not belong to this group, its content hash is
    /// empty, or `location` is already registered under a different key.
    pub fn insert(
        &mut self,
        key: CacheKey,
        location: CacheBlockLocation,
    ) -> Option<CacheBlockLocation> {
        self.validate_key(&key);
        if let Some(idx) = self.by_location.get(&location).copied() {
            assert!(
                self.entries[idx].key == key,
                "one cache block location cannot change cache key"
            );
            let epoch = self.next_epoch();
            self.entries[idx].last_access_epoch = epoch;
            return None;
        }
        if let Some(idx) = self.by_key.get(&key).copied() {
            let canonical = self.entries[idx].location;
            let epoch = self.next_epoch();
            self.entries[idx].last_access_epoch = epoch;
            return Some(canonical);
        }
        let idx = self.entries.len();
        let epoch = self.next_epoch();
        self.entries.push(CacheEntry {
            key: key.clone(),
            location,
            last_access_epoch: epoch,
        });
        self.by_key.insert(key, idx);
        self.by_location.insert(location, idx);
        None
    }

    /// Removes the entry for `key`, returning its location.
    pub fn remove(&mut self, key: &CacheKey) -> Option<CacheBlockLocation> {
        let idx = self.by_key.get(key).copied()?;
        let location = self.entries[idx].location;
        self.remove_at(idx);
        Some(location)
    }

    /// Removes the entry at `location`, returning its key.
    pub fn remove_location(&mut self, location: CacheBlockLocation) -> Option<CacheKey> {
        let idx = self.by_location.get(&location).copied()?;
        let key = self.entries[idx].key.clone();
        self.remove_at(idx);
        Some(key)
    }

    /// Looks up the canonical location for `key`, recording the access (LRU
    /// touch): the entry's `last_access_epoch` advances so
    /// [`PrefixCacheIndex::evict_oldest`] prefers colder entries.
    pub fn query(&mut self, key: &CacheKey) -> Option<CacheBlockLocation> {
        let idx = self.by_key.get(key).copied()?;
        let epoch = self.next_epoch();
        let entry = &mut self.entries[idx];
        entry.last_access_epoch = epoch;
        Some(entry.location)
    }

    /// Whether `key` is cached (immutable; used by the matcher).
    #[must_use]
    pub fn contains(&self, key: &CacheKey) -> bool {
        self.by_key.contains_key(key)
    }

    /// Whether `location` is a cached entry.
    #[must_use]
    pub fn contains_location(&self, location: CacheBlockLocation) -> bool {
        self.by_location.contains_key(&location)
    }

    /// The key registered at `location`, if any.
    #[must_use]
    pub fn key_for(&self, location: CacheBlockLocation) -> Option<&CacheKey> {
        self.by_location
            .get(&location)
            .map(|&idx| &self.entries[idx].key)
    }

    /// The most recent access epoch of `key`, if cached.
    #[must_use]
    pub fn last_access_epoch(&self, key: &CacheKey) -> Option<u64> {
        self.by_key
            .get(key)
            .map(|&idx| self.entries[idx].last_access_epoch)
    }

    /// Number of cached entries.
    #[must_use]
    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }

    /// Evicts the entry with the oldest `last_access_epoch` (LRU) and returns
    /// its `(key, location)`, or `None` when the index is empty.
    pub fn evict_oldest(&mut self) -> Option<(CacheKey, CacheBlockLocation)> {
        let coldest = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, entry)| entry.last_access_epoch)?
            .0;
        let key = self.entries[coldest].key.clone();
        let location = self.entries[coldest].location;
        self.remove_at(coldest);
        Some((key, location))
    }

    fn next_epoch(&mut self) -> u64 {
        let epoch = self.next_epoch;
        self.next_epoch += 1;
        epoch
    }

    fn validate_key(&self, key: &CacheKey) {
        assert!(
            key.group_id == self.group_id,
            "cache key group does not match index"
        );
        assert!(
            !key.content_hash.is_empty(),
            "cache key content hash must not be empty"
        );
    }

    /// Removes `index`, repairing the map indices of the element moved into
    /// the vacated slot by `swap_remove`.
    fn remove_at(&mut self, index: usize) {
        let key = self.entries[index].key.clone();
        let location = self.entries[index].location;
        self.by_key.remove(&key);
        self.by_location.remove(&location);
        let last = self.entries.len() - 1;
        if index != last {
            self.entries.swap_remove(index);
            let moved = &self.entries[index];
            self.by_key.insert(moved.key.clone(), index);
            self.by_location.insert(moved.location, index);
        } else {
            self.entries.pop();
        }
    }
}

// ---------------------------------------------------------------------------
// PrefixMatcher
// ---------------------------------------------------------------------------

/// Full-attention prefix matcher: a match is a contiguous, hole-free run of
/// cached pages starting at the probe position.
///
/// The probe walks a request's page keys left-to-right and stops at the first
/// miss (an uncached page is a hole and counts as a miss), so the reported
/// run is contiguous by construction — a shorter boundary always remains
/// valid.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrefixMatcher;

impl PrefixMatcher {
    /// Probes `keys[begin_blocks..end_blocks)` where
    /// `end_blocks = min(max_blocks, keys.len())` — `max_blocks` is an
    /// absolute end, matching the upstream C++ semantics
    /// (`end = min(keys.len(), max(0, max_blocks))`).
    ///
    /// Returns the contiguous hit run aligned to `begin_blocks`: `hits[i]` is
    /// `1` when `keys[begin_blocks + i]` is cached, and the run stops at the
    /// first miss, so `hits` is all ones and never longer than the probed
    /// range.
    #[must_use]
    pub fn probe(
        &self,
        index: &PrefixCacheIndex,
        keys: &[CacheKey],
        begin_blocks: usize,
        max_blocks: usize,
    ) -> Vec<u8> {
        let end_blocks = keys.len().min(max_blocks);
        let mut hits = Vec::with_capacity(end_blocks.saturating_sub(begin_blocks));
        for key in keys.iter().take(end_blocks).skip(begin_blocks) {
            if !index.contains(key) {
                break;
            }
            hits.push(1);
        }
        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- PrefixHasher --------------------------------------------------------

    // Golden vectors computed with an independent SHA-256 implementation
    // (.NET System.Security.Cryptography) by the upstream test suite.

    #[test]
    fn empty_page_with_empty_prior() {
        assert_eq!(
            hash_prefix_page(&[], "", &[]),
            "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"
        );
    }

    #[test]
    fn tokens_with_empty_prior() {
        assert_eq!(
            hash_prefix_page(&[1, 2, 3], "", &[]),
            "a452f93a8b397e453162a0ee3b3408c00b5ddb4587f936b4ce2b66659feaedaf"
        );
    }

    #[test]
    fn chained_page_absorbs_prior_digest() {
        let prior = "a452f93a8b397e453162a0ee3b3408c00b5ddb4587f936b4ce2b66659feaedaf";
        assert_eq!(
            hash_prefix_page(&[4, 5], prior, &[]),
            "37a58e214fcc09dceb07aa0f4ec9b1f8e644e9b5c855c8f0725d37749f9c4386"
        );
    }

    #[test]
    fn extra_keys_are_framed_after_tokens() {
        assert_eq!(
            hash_prefix_page(&[1], "", &["loraA", "b"]),
            "0139d024eb5c28ad07c2b3dfc4b05aca2fa8d80155b0ffe853cf7e19bea47130"
        );
    }

    #[test]
    fn negative_tokens_are_two_complement_u32() {
        // tokens [-1] == u32 0xFFFFFFFF: [prior_len 0][count 1][0xFFFFFFFF le]
        let mut ctx = Sha256::new();
        ctx.update([0u8, 0, 0, 0]);
        ctx.update([1u8, 0, 0, 0]);
        ctx.update([0xFF, 0xFF, 0xFF, 0xFF]);
        let expected = digest_to_hex(&ctx.finalize());
        assert_eq!(hash_prefix_page(&[-1], "", &[]), expected);
    }

    #[test]
    fn hex_round_trips() {
        let digest = hash_prefix_page(&[7, 8], "", &[]);
        assert_eq!(digest_to_hex(&hex_to_bytes(&digest)), digest);
        assert_eq!(digest.len(), 64);
        // Uppercase input decodes to the same bytes.
        assert_eq!(hex_to_bytes(&digest.to_uppercase()), hex_to_bytes(&digest));
    }

    #[test]
    fn compute_prefix_hashes_chains_pages() {
        let pages: Vec<&[i32]> = vec![&[1, 2, 3], &[4, 5]];
        let hashes = compute_prefix_hashes(&pages, "", &[]);
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes[0], hash_prefix_page(&[1, 2, 3], "", &[]));
        assert_eq!(hashes[1], hash_prefix_page(&[4, 5], &hashes[0], &[]));
    }

    #[test]
    fn compute_prefix_hashes_applies_per_page_extra_keys() {
        let pages: Vec<&[i32]> = vec![&[1], &[2]];
        let extras: Vec<&[&str]> = vec![&["loraA"], &[]];
        let hashes = compute_prefix_hashes(&pages, "", &extras);
        assert_eq!(hashes[0], hash_prefix_page(&[1], "", &["loraA"]));
        assert_eq!(hashes[1], hash_prefix_page(&[2], &hashes[0], &[]));
    }

    #[test]
    fn advance_prefix_hashes_returns_subrange() {
        let pages: Vec<&[i32]> = vec![&[1], &[2], &[3]];
        let all = compute_prefix_hashes(&pages, "", &[]);
        let tail = advance_prefix_hashes(&pages, 1, &all[0], 3);
        assert_eq!(tail, vec![all[1].clone(), all[2].clone()]);
    }

    #[test]
    #[should_panic(expected = "hash range must be non-empty")]
    fn advance_prefix_hashes_rejects_empty_range() {
        advance_prefix_hashes(&[&[1]], 0, "", 0);
    }

    // -- PrefixCacheIndex ----------------------------------------------------

    fn key(group: u32, hash: &str, offset: i32) -> CacheKey {
        CacheKey {
            namespace_id: 0,
            group_id: group,
            content_hash: hash.to_string(),
            page_offset: offset,
        }
    }

    fn loc(block: i32, slot: i32) -> CacheBlockLocation {
        CacheBlockLocation {
            lcm_block_id: block,
            slot_index: slot,
        }
    }

    fn page_hash(tokens: &[i32]) -> String {
        hash_prefix_page(tokens, "", &[])
    }

    #[test]
    fn insert_query_remove_round_trip() {
        let mut index = PrefixCacheIndex::new(1);
        assert_eq!(index.insert(key(1, "h1", 0), loc(3, 0)), None);
        assert_eq!(index.num_entries(), 1);
        assert!(index.contains(&key(1, "h1", 0)));
        assert!(index.contains_location(loc(3, 0)));
        assert_eq!(index.key_for(loc(3, 0)), Some(&key(1, "h1", 0)));
        assert_eq!(index.query(&key(1, "h1", 0)), Some(loc(3, 0)));
        assert_eq!(index.remove(&key(1, "h1", 0)), Some(loc(3, 0)));
        assert_eq!(index.num_entries(), 0);
        assert!(!index.contains(&key(1, "h1", 0)));
        assert!(index.query(&key(1, "h1", 0)).is_none());
    }

    #[test]
    fn insert_same_key_keeps_canonical_location() {
        let mut index = PrefixCacheIndex::new(0);
        assert_eq!(index.insert(key(0, "h", 0), loc(0, 0)), None);
        // A duplicate page with the same content is reported as a duplicate:
        // the caller releases it and keeps the canonical location.
        assert_eq!(index.insert(key(0, "h", 0), loc(9, 9)), Some(loc(0, 0)));
        assert_eq!(index.num_entries(), 1);
        assert_eq!(index.query(&key(0, "h", 0)), Some(loc(0, 0)));
        assert!(!index.contains_location(loc(9, 9)));
    }

    #[test]
    #[should_panic(expected = "one cache block location cannot change cache key")]
    fn insert_same_location_different_key_panics() {
        let mut index = PrefixCacheIndex::new(0);
        index.insert(key(0, "h1", 0), loc(0, 0));
        index.insert(key(0, "h2", 0), loc(0, 0));
    }

    #[test]
    fn remove_repairs_swap_remove_indices() {
        let mut index = PrefixCacheIndex::new(0);
        for (i, h) in ["a", "b", "c", "d"].iter().enumerate() {
            assert_eq!(index.insert(key(0, h, i as i32), loc(i as i32, 0)), None);
        }
        // Removing the first entry swaps "d" into slot 0; both secondary
        // indices must be repaired so every remaining key/location resolves.
        assert_eq!(index.remove(&key(0, "a", 0)), Some(loc(0, 0)));
        assert_eq!(index.num_entries(), 3);
        for (i, h) in ["b", "c", "d"].iter().enumerate() {
            let idx = i + 1;
            assert_eq!(
                index.query(&key(0, h, idx as i32)),
                Some(loc(idx as i32, 0))
            );
            assert_eq!(
                index.key_for(loc(idx as i32, 0)),
                Some(&key(0, h, idx as i32))
            );
        }
        assert!(!index.contains(&key(0, "a", 0)));
        assert!(!index.contains_location(loc(0, 0)));
        // Removing the moved tail entry also works.
        assert_eq!(index.remove_location(loc(3, 0)), Some(key(0, "d", 3)));
        assert_eq!(index.num_entries(), 2);
    }

    #[test]
    fn evict_oldest_uses_last_access_epoch() {
        let mut index = PrefixCacheIndex::new(0);
        index.insert(key(0, "a", 0), loc(1, 1));
        index.insert(key(0, "b", 0), loc(2, 2));
        index.insert(key(0, "c", 0), loc(3, 3));
        // Touch "a" so "b" becomes the coldest entry.
        assert_eq!(index.query(&key(0, "a", 0)), Some(loc(1, 1)));
        assert!(
            index.last_access_epoch(&key(0, "a", 0)) > index.last_access_epoch(&key(0, "b", 0))
        );
        assert_eq!(index.evict_oldest(), Some((key(0, "b", 0), loc(2, 2))));
        assert!(!index.contains(&key(0, "b", 0)));
        assert!(index.contains(&key(0, "a", 0)));
        assert!(index.contains(&key(0, "c", 0)));
    }

    // -- PrefixMatcher -------------------------------------------------------

    #[test]
    fn full_attn_probe_stops_at_first_miss() {
        let mut index = PrefixCacheIndex::new(0);
        for tokens in [[1], [2], [4], [5]] {
            index.insert(key(0, &page_hash(&tokens), tokens[0]), loc(tokens[0], 0));
        }
        let keys = vec![
            key(0, &page_hash(&[1]), 1),
            key(0, &page_hash(&[2]), 2),
            key(0, &page_hash(&[3]), 3), // uncached -> hole
            key(0, &page_hash(&[4]), 4),
        ];
        let matcher = PrefixMatcher;
        // Hits pages [1], [2]; [3] is a miss so the run stops and [4] is
        // never probed.
        assert_eq!(matcher.probe(&index, &keys, 0, 4), vec![1, 1]);
    }

    #[test]
    fn full_attn_probe_respects_begin_and_max() {
        let mut index = PrefixCacheIndex::new(0);
        for tokens in [[1], [2], [3]] {
            index.insert(key(0, &page_hash(&tokens), tokens[0]), loc(tokens[0], 0));
        }
        let keys = vec![
            key(0, &page_hash(&[1]), 1),
            key(0, &page_hash(&[2]), 2),
            key(0, &page_hash(&[3]), 3),
        ];
        let matcher = PrefixMatcher;
        // max_blocks is an absolute end: begin=1, max=2 probes only keys[1].
        assert_eq!(matcher.probe(&index, &keys, 1, 2), vec![1]);
        assert_eq!(matcher.probe(&index, &keys, 1, 3), vec![1, 1]);
        // max=0 or begin==end probes nothing.
        assert_eq!(matcher.probe(&index, &keys, 0, 0), Vec::<u8>::new());
        assert_eq!(matcher.probe(&index, &keys, 3, 3), Vec::<u8>::new());
    }

    #[test]
    fn full_attn_probe_treats_hole_as_miss() {
        let mut index = PrefixCacheIndex::new(0);
        index.insert(key(0, &page_hash(&[1]), 1), loc(1, 0));
        index.insert(key(0, &page_hash(&[3]), 3), loc(3, 0));
        let keys = vec![
            key(0, &page_hash(&[1]), 1),
            key(0, &page_hash(&[2]), 2),
            key(0, &page_hash(&[3]), 3),
        ];
        let matcher = PrefixMatcher;
        // A hole at keys[1] cuts the run even though keys[2] is cached.
        assert_eq!(matcher.probe(&index, &keys, 0, 3), vec![1]);
    }
}
