using System;

namespace Generated
{
    public class BetBuilderLeg14Entity
    {
        public long Id { get; set; }
        public long TransId { get; set; }
        public long RefNo { get; set; }
        public int MatchMarketSelectionId { get; set; }
        public required string Status { get; set; }
        public int MarketTypeId { get; set; }
        public int SelectionTypeId { get; set; }
        public decimal? Point { get; set; }
        public required string MatchMarketDetails { get; set; }
        public required string MatchSelectionDetails { get; set; }
        public int CustomerId { get; set; }
        public DateTime TransDate { get; set; }
        public bool? Ruben { get; set; }
        public DateTime LastModifiedOn { get; set; }
    }
}
