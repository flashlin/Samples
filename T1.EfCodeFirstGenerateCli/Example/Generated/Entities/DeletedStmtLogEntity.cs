using System;

namespace Generated
{
    public class DeletedStmtLogEntity
    {
        public long transid { get; set; }
        public long refno { get; set; }
        public int custid { get; set; }
        public DateTime transdate { get; set; }
        public required string status { get; set; }
        public required string winlost { get; set; }
        public required string creator { get; set; }
        public DateTime? winlostdate { get; set; }
        public required string betfrom { get; set; }
        public required string betcheck { get; set; }
        public DateTime? checktime { get; set; }
        public required string actualrate { get; set; }
        public int? recommend { get; set; }
        public int? mrecommend { get; set; }
        public int srecommend { get; set; }
        public byte? ruben { get; set; }
        public byte? bettype { get; set; }
        public byte currency { get; set; }
        public required string actual_stake { get; set; }
        public required string transdesc { get; set; }
        public required string ip { get; set; }
        public required string username { get; set; }
        public required string currencystr { get; set; }
        public int? betstatus { get; set; }
        public required string creatorName { get; set; }
    }
}
