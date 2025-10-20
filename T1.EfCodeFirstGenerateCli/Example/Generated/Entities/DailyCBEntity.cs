using System;

namespace Generated
{
    public class DailyCBEntity
    {
        public int CBID { get; set; }
        public int? ProductType { get; set; }
        public int? SubProductType { get; set; }
        public DateTime? CBDate { get; set; }
        public int? FromID { get; set; }
        public required string FromAccount { get; set; }
        public int? ToID { get; set; }
        public required string ToAccount { get; set; }
        public decimal Amount { get; set; }
        public int? TxnType { get; set; }
        public long? RefNo { get; set; }
        public required string Description { get; set; }
        public DateTime? CreatedOn { get; set; }
        public required string CreatedBy { get; set; }
    }
}
