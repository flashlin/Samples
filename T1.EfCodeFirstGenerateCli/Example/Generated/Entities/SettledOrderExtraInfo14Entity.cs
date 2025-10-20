using System;

namespace Generated
{
    public class SettledOrderExtraInfo14Entity
    {
        public Guid Id { get; set; }
        public long transid { get; set; }
        public long refno { get; set; }
        public DateTime transdate { get; set; }
        public int? oddsid { get; set; }
        public decimal? hdp1 { get; set; }
        public decimal? hdp2 { get; set; }
        public decimal? odds { get; set; }
        public required string status { get; set; }
        public int? livehomescore { get; set; }
        public int? liveawayscore { get; set; }
        public bool? liveindicator { get; set; }
        public required string betteam { get; set; }
        public required string comstatus { get; set; }
        public DateTime? winlostdate { get; set; }
        public required string betfrom { get; set; }
        public DateTime? checktime { get; set; }
        public decimal? actualrate { get; set; }
        public required string modds { get; set; }
        public byte? bettype { get; set; }
        public required string ip { get; set; }
        public required string username { get; set; }
        public int? oddsstyle { get; set; }
        public int? betstatus { get; set; }
        public required string creatorname { get; set; }
        public byte? DangerLevel { get; set; }
        public decimal? BlindRiskRate { get; set; }
        public int? DirectCustId { get; set; }
        public int? MatchResultId { get; set; }
        public int? NewBetType { get; set; }
        public int? DisplayType { get; set; }
        public required string BetCondition { get; set; }
        public int? BetTypeGroupId { get; set; }
        public int? MemberStatus { get; set; }
        public int? FinalHomeScore { get; set; }
        public int? FinalAwayScore { get; set; }
        public int? HTHomeScore { get; set; }
        public int? HTAwayScore { get; set; }
        public DateTime? EventDate { get; set; }
        public int? SportId { get; set; }
        public int? LeagueId { get; set; }
        public int? HomeId { get; set; }
        public int? AwayId { get; set; }
        public int? MatchId { get; set; }
        public required string Result { get; set; }
        public DateTime? KickOffTime { get; set; }
        public required string creator { get; set; }
        public required string betcheck { get; set; }
        public required string oddsspread { get; set; }
        public long? betdaqid { get; set; }
        public byte? statuswinlost { get; set; }
        public byte? currency { get; set; }
        public required string transdesc { get; set; }
        public required byte[] tstamp { get; set; }
        public required string currencystr { get; set; }
        public required string CountryCode { get; set; }
        public int? TraderID { get; set; }
        public byte? betpage { get; set; }
    }
}
