use std::sync::atomic::Ordering;

use super::{INITIAL_CAPACITY, SlotMeta, SuffixBag, WIDTH};

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

        for slot in 0..WIDTH {
            let Some(bytes) = self.get(slot) else {
                continue;
            };

            let new_offset: usize = new_data.len();
            new_data.extend_from_slice(bytes);

            #[expect(clippy::cast_possible_truncation)]
            {
                new_slots[slot] = SlotMeta {
                    offset: new_offset as u32,
                    len: bytes.len() as u16,
                    _pad: 0,
                };
            }
        }

        // Force a fresh count in the clone regardless of a concurrent clear.
        let cloned: Self = Self::from_parts(new_data, new_slots);
        cloned.suffix_count.store(
            new_slots.iter().filter(|m| m.has_suffix()).count() as u8,
            Ordering::Relaxed,
        );

        cloned
    }
}
