using System;

namespace Generated
{
    public class MpPTPresetEntity
    {
        public int CustomerId { get; set; }
        public int ParentId { get; set; }
        public decimal MinimumPT { get; set; }
        public decimal ForcedPT { get; set; }
        public bool TakeRemaining { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime? ModifiedDate { get; set; }
    }
}
