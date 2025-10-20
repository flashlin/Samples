using System;

namespace Generated
{
    public class PresetPositionTakingEntity
    {
        public int CustomerId { get; set; }
        public int ParentId { get; set; }
        public byte Type { get; set; }
        public decimal ParentMin { get; set; }
        public decimal ParentForce { get; set; }
        public bool TakeRemaining { get; set; }
        public DateTime? ModifiedOn { get; set; }
        public required string ModifiedBy { get; set; }
    }
}
