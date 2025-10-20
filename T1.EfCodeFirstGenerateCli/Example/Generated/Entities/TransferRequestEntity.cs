using System;

namespace Generated
{
    public class TransferRequestEntity
    {
        public int requestid { get; set; }
        public long requestrefno { get; set; }
        public DateTime requestdate { get; set; }
        public int custid { get; set; }
        public decimal? amount { get; set; }
        public bool istakeremaining { get; set; }
        public required string fromproduct { get; set; }
        public required string toproduct { get; set; }
        public int requesterid { get; set; }
        public required string requestername { get; set; }
        public required string status { get; set; }
        public required string remark { get; set; }
        public int? Mode { get; set; }
        public int Wonglaitype { get; set; }
    }
}
