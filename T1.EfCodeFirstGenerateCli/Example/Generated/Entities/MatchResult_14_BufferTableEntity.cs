using System;

namespace Generated
{
    public class MatchResult_14_BufferTableEntity
    {
        public int MatchResultId { get; set; }
        public int MatchId { get; set; }
        public int LeagueId { get; set; }
        public int HomeId { get; set; }
        public int AwayId { get; set; }
        public int BetTypeGroupId { get; set; }
        public DateTime? EventDate { get; set; }
        public required string EventStatus { get; set; }
        public int? LiveHomeScore { get; set; }
        public int? LiveAwayScore { get; set; }
        public int? FinalHomeScore { get; set; }
        public int? FinalAwayScore { get; set; }
        public int Creator { get; set; }
        public required string MatchCode { get; set; }
        public DateTime? KickOffTime { get; set; }
        public required string ShowTime { get; set; }
        public int? HTHomeScore { get; set; }
        public int? HTAwayScore { get; set; }
        public byte? Ruben { get; set; }
        public bool? Multiple { get; set; }
        public required byte[] tstamp { get; set; }
        public int? SportId { get; set; }
        public required string Result { get; set; }
        public required string Color { get; set; }
        public required string Remark { get; set; }
        public byte? OtherStatus { get; set; }
        public int? OtherStatus2 { get; set; }
        public DateTime? AbsKickoffTime { get; set; }
        public byte EventType { get; set; }
        public int? EventTypeID { get; set; }
        public int? LiveHdpTrader { get; set; }
        public int? LiveOuTrader { get; set; }
        public int? HdpTrader { get; set; }
        public int? OuTrader { get; set; }
        public int? ft1x2Trader { get; set; }
        public int? fh1x2Trader { get; set; }
        public int? ShowTimeDisplayType { get; set; }
        public int? LiveFhHdpTrader { get; set; }
        public int? LiveFhOuTrader { get; set; }
        public DateTime? CreateOn { get; set; }
        public required string HomeJersey { get; set; }
        public required string AwayJersey { get; set; }
        public int ParentId { get; set; }
        public DateTime LastModifiedOn { get; set; }
        public required string Channel { get; set; }
        public bool? ScoreVerified { get; set; }
        public int? LiveCsTrader { get; set; }
        public int? LiveFhCsTrader { get; set; }
        public DateTime OwnTstamp { get; set; }
    }
}
