using System;

namespace Generated
{
    public class match14Entity
    {
        public int matchid { get; set; }
        public int leagueid { get; set; }
        public int homeid { get; set; }
        public int awayid { get; set; }
        public DateTime? eventdate { get; set; }
        public required string eventstatus { get; set; }
        public required string betstatus { get; set; }
        public int? livehomescore { get; set; }
        public int? liveawayscore { get; set; }
        public int? finalhomescore { get; set; }
        public int? finalawayscore { get; set; }
        public int creator { get; set; }
        public required string matchcode { get; set; }
        public DateTime? kickofftime { get; set; }
        public DateTime? closedtime { get; set; }
        public required string showtime { get; set; }
        public int? hthomescore { get; set; }
        public int? htawayscore { get; set; }
        public decimal? hometotal { get; set; }
        public decimal? awaytotal { get; set; }
        public byte? ruben { get; set; }
        public bool? multiple { get; set; }
        public required byte[] tstamp { get; set; }
        public int? sportid { get; set; }
        public byte? tlive { get; set; }
        public int? livecontrolhdp { get; set; }
        public int? livecontrolou { get; set; }
        public required string channel { get; set; }
        public required string result { get; set; }
        public required string color { get; set; }
        public required string remark { get; set; }
        public byte? otherstatus { get; set; }
        public int? otherstatus2 { get; set; }
        public DateTime? absKickoffTime { get; set; }
        public byte? EventType { get; set; }
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
        public int? rid { get; set; }
        public int ParentId { get; set; }
        public DateTime? LastModifiedOn { get; set; }
        public bool? ScoreVerified { get; set; }
        public int? LiveCsTrader { get; set; }
        public int? LiveFhCsTrader { get; set; }
    }
}
