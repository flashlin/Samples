using System;

namespace Generated
{
    public class MpPlayerBetSettingEntity
    {
        public int CustomerId { get; set; }
        public int ParentId { get; set; }
        public decimal? Credit { get; set; }
        public decimal? MinimumBet { get; set; }
        public decimal? MaximumBet { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime? ModifiedDate { get; set; }
    }
}
