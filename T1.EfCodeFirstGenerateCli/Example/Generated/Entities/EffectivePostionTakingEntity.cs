using System;

namespace Generated
{
    public class EffectivePostionTakingEntity
    {
        public int PlayerId { get; set; }
        public int AgentId { get; set; }
        public byte Type { get; set; }
        public decimal ParentMin { get; set; }
        public decimal ParentForce { get; set; }
        public bool TakeRemaining { get; set; }
        public decimal Effective { get; set; }
        public DateTime? ModifiedOn { get; set; }
        public required string ModifiedBy { get; set; }
    }
}
