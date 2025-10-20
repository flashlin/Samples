using System;

namespace Generated
{
    public class AutoSettlementEntity
    {
        public int MatchResultId { get; set; }
        public bool? IsHTReadyForAutoSettle { get; set; }
        public bool? ISFTReadyForAutoSettle { get; set; }
        public int? HTHomeScore { get; set; }
        public int? HTAwayScore { get; set; }
        public int? FinalHomeScore { get; set; }
        public int? FinalAwayScore { get; set; }
        public required string FGLG { get; set; }
        public DateTime? EventDate { get; set; }
        public int? SettlementStatus { get; set; }
        public required string ErrorCode { get; set; }
        public required string Provider { get; set; }
        public required string ProcessingStatus { get; set; }
        public int? SportId { get; set; }
        public required string SettlementAction { get; set; }
        public DateTime? CreatedOn { get; set; }
        public DateTime? LastModifiedOn { get; set; }
    }
}
