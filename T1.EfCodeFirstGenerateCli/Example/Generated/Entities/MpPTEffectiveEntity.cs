using System;

namespace Generated
{
    public class MpPTEffectiveEntity
    {
        public int CustomerId { get; set; }
        public int ParentId { get; set; }
        public int RoleId { get; set; }
        public decimal MinimumPT { get; set; }
        public decimal ForcedPT { get; set; }
        public bool TakeRemaining { get; set; }
        public decimal PT { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime? ModifiedDate { get; set; }
    }
}
