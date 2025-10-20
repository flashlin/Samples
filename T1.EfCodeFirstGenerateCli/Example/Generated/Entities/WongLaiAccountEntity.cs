using System;

namespace Generated
{
    public class WongLaiAccountEntity
    {
        public int Currency { get; set; }
        public byte Type { get; set; }
        public required string CurrencyStr { get; set; }
        public required string TypeStr { get; set; }
        public int CustID { get; set; }
        public int AgentID { get; set; }
        public int MaID { get; set; }
        public int SmaID { get; set; }
        public required string UserName { get; set; }
        public int status { get; set; }
        public DateTime? tstamp { get; set; }
    }
}
