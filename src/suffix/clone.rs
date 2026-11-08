use std::sync::atomic::Ordering;

use super::{INITIAL_CAPACITY, SlotMeta, SuffixBag, WIDTH};

#[expect(
    clippy::cast_possible_truncation,
    reason = "at most WIDTH (15) suffixes fit in a u8"
)]
impl Clone for SuffixBag {
    /// Clone with compaction, only copies active suffix data.
    ///
    /// This avoids copying dead/garbage data from the original buffer.
    fn clone(&self) -> Self {
        if self.count() == 0 {
            return Self::new();
        }

        // Calculate exact size needed for active suffixes.
        let needed: usize = (0..WIDTH)
            .map(|slot: usize| self.load_meta(slot))
            .filter(|meta: &SlotMeta| meta.has_suffix())
            .map(|meta: SlotMeta| meta.len as usize)
            .sum();

        let capacity: usize = needed.next_power_of_two().max(INITIAL_CAPACITY);
        let mut new_data: Vec<u8> = Vec::with_capacity(capacity);
        let mut new_slots: [SlotMeta; WIDTH] = [SlotMeta::EMPTY; WIDTH];

        for (slot, new_slot) in new_slots.iter_mut().enumerate() {
            let Some(bytes) = self.get(slot) else {
                continue;
            };

            let new_offset: usize = new_data.len();
            new_data.extend_from_slice(bytes);

            #[expect(clippy::cast_possible_truncation)]
            {
                *new_slot = SlotMeta {
                    offset: new_offset as u32,
                    len: bytes.len() as u16,
                    pad: 0,
                };
            }
        }

        // Force a fresh count in the clone regardless of a concurrent clear.
        let cloned: Self = Self::from_parts(new_data, new_slots);
        let count: u8 = new_slots.iter().filter(|m| m.has_suffix()).count() as u8;
        cloned.suffix_count.store(count, Ordering::Relaxed);

        cloned
    }
}
