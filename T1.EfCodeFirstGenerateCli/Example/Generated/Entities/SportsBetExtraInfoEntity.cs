using System;

namespace Generated
{
    public class SportsBetExtraInfoEntity
    {
        public long TransId { get; set; }
        public int LeoEnumValue { get; set; }
        public DateTime LastModifiedOn { get; set; }
        public int? MarketTypeId { get; set; }
        public long? SelectionId { get; set; }
        public int? SelectionTypeId { get; set; }
        public int? MatchStatTypeId { get; set; }
        public required string MarketDetails { get; set; }
        public required string SelectionDetails { get; set; }
        public required string BetTypeName { get; set; }
        public long? TraceBetId { get; set; }
    }
}
