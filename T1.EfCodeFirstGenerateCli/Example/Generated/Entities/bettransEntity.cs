using System;

namespace Generated
{
    public class bettransEntity
    {
        public long transid { get; set; }
        public long refno { get; set; }
        public int custid { get; set; }
        public DateTime transdate { get; set; }
        public int? oddsid { get; set; }
        public decimal? hdp1 { get; set; }
        public decimal? hdp2 { get; set; }
        public decimal? odds { get; set; }
        public decimal? stake { get; set; }
        public required string status { get; set; }
        public decimal? winlost { get; set; }
        public int? livehomescore { get; set; }
        public int? liveawayscore { get; set; }
        public bool? liveindicator { get; set; }
        public required string betteam { get; set; }
        public required string creator { get; set; }
        public required string comstatus { get; set; }
        public DateTime? winlostdate { get; set; }
        public required string betfrom { get; set; }
        public required string betcheck { get; set; }
        public DateTime? checktime { get; set; }
        public required string oddsspread { get; set; }
        public decimal? apositiontaking { get; set; }
        public decimal? mpositiontaking { get; set; }
        public decimal? tpositiontaking { get; set; }
        public decimal? awinlost { get; set; }
        public decimal? mwinlost { get; set; }
        public decimal? playerdiscount { get; set; }
        public decimal? discount { get; set; }
        public decimal? adiscount { get; set; }
        public decimal? playercomm { get; set; }
        public decimal? comm { get; set; }
        public decimal? acomm { get; set; }
        public decimal? actualrate { get; set; }
        public int? matchid { get; set; }
        public required string modds { get; set; }
        public int? recommend { get; set; }
        public int? mrecommend { get; set; }
        public long? betdaqid { get; set; }
        public byte? ruben { get; set; }
        public byte? statuswinlost { get; set; }
        public byte? bettype { get; set; }
        public byte currency { get; set; }
        public decimal? actual_stake { get; set; }
        public required string transdesc { get; set; }
        public required byte[] tstamp { get; set; }
        public required string ip { get; set; }
        public decimal? sdiscount { get; set; }
        public decimal? scomm { get; set; }
        public decimal? spositiontaking { get; set; }
        public decimal? swinlost { get; set; }
        public int srecommend { get; set; }
        public required string username { get; set; }
        public required string currencystr { get; set; }
        public int? oddsstyle { get; set; }
        public int? betstatus { get; set; }
        public required string creatorname { get; set; }
        public int? sportid { get; set; }
        public int? leagueid { get; set; }
        public byte? DangerLevel { get; set; }
        public decimal? BlindRiskRate { get; set; }
        public required string CountryCode { get; set; }
        public int? DirectCustId { get; set; }
        public int? MatchResultId { get; set; }
        public int? NewBetType { get; set; }
        public int? DisplayType { get; set; }
        public required string BetCondition { get; set; }
        public int? BetTypeGroupId { get; set; }
        public int? MemberStatus { get; set; }
        public int? TraderID { get; set; }
        public byte? betpage { get; set; }
        public DateTime? OwnTstamp { get; set; }
        public DateTime? LastModifiedOn { get; set; }
        public DateTime? CreatedOn { get; set; }
        public DateTime? SettlementTime { get; set; }
        public long? originalid { get; set; }
        public long ID { get; set; }
        public decimal? CommissionableStake { get; set; }
    }
}
