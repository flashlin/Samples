using System;

namespace Generated
{
    public class AutoSettlementRecordEntity
    {
        public int Id { get; set; }
        public int MatchResultId { get; set; }
        public required string Result { get; set; }
        public required string ErrorCode { get; set; }
        public required string Provider { get; set; }
        public required string ProcessingStatus { get; set; }
        public int? SportId { get; set; }
        public required string SettlementAction { get; set; }
        public DateTime? CreatedOn { get; set; }
        public DateTime? LastModifiedOn { get; set; }
        public DateTime EventDate { get; set; }
    }
}
