using System;

namespace Generated
{
    public class MoneyTransferLogEntity
    {
        public int MTID { get; set; }
        public byte TransferType { get; set; }
        public int? FromID { get; set; }
        public required string FromAccountID { get; set; }
        public int? ToID { get; set; }
        public required string ToAccountID { get; set; }
        public decimal Amount { get; set; }
        public required string ISOCurrency { get; set; }
        public decimal? MarketRate { get; set; }
        public byte TransferStatus { get; set; }
        public bool PaymentMethod { get; set; }
        public required string Description { get; set; }
        public required string Remark { get; set; }
        public required string CreatedBy { get; set; }
        public DateTime? CreatedOn { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime? ModifiedOn { get; set; }
        public bool? IsRead { get; set; }
        public int MTBID { get; set; }
        public required string TransferFollowupGroup { get; set; }
    }
}
