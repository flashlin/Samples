using System;

namespace Generated
{
    public class MpAgentsBetSettingEntity
    {
        public int CustomerId { get; set; }
        public int ParentId { get; set; }
        public decimal? Credit { get; set; }
        public decimal? MaxCredit { get; set; }
        public decimal? MinimumBet { get; set; }
        public decimal? MaximumBet { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime? ModifiedDate { get; set; }
    }
}
