using System;

namespace Generated
{
    public class bettransmEntity
    {
        public int transid { get; set; }
        public int custid { get; set; }
        public DateTime transdate { get; set; }
        public int oddsid { get; set; }
        public decimal? hdp1 { get; set; }
        public decimal? hdp2 { get; set; }
        public decimal? odds { get; set; }
        public required string status { get; set; }
        public int? livehomescore { get; set; }
        public int? liveawayscore { get; set; }
        public bool? liveindicator { get; set; }
        public required string betteam { get; set; }
        public long refno { get; set; }
        public required string comstatus { get; set; }
        public DateTime? winlostdate { get; set; }
        public required string betcheck { get; set; }
        public decimal? tpositiontaking { get; set; }
        public int? matchid { get; set; }
        public DateTime? matchdate { get; set; }
        public decimal? finalodds { get; set; }
        public required string isfinish { get; set; }
        public byte? statuswinlost { get; set; }
        public byte? ruben { get; set; }
        public byte? bettype { get; set; }
        public int? srecommend { get; set; }
        public decimal? sdiscount { get; set; }
        public required byte[] tstamp { get; set; }
        public int? sportid { get; set; }
        public int? MatchResultId { get; set; }
        public int? NewBetType { get; set; }
        public int? DisplayType { get; set; }
        public int? BetTypeGroupId { get; set; }
        public DateTime? OwnTstamp { get; set; }
        public DateTime? CheckTime { get; set; }
        public required string BetCondition { get; set; }
        public long ID { get; set; }
    }
}
