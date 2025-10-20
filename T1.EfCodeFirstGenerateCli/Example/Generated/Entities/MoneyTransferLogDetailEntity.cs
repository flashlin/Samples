using System;

namespace Generated
{
    public class MoneyTransferLogDetailEntity
    {
        public int MTDID { get; set; }
        public int MTID { get; set; }
        public byte TransferDetailType { get; set; }
        public int? FromID { get; set; }
        public required string FromAccountID { get; set; }
        public int? ToID { get; set; }
        public required string ToAccountID { get; set; }
        public decimal Amount { get; set; }
        public decimal? MarketRate { get; set; }
        public byte TransferDetailStatus { get; set; }
        public required string Description { get; set; }
        public required string Remark { get; set; }
        public required string CreatedBy { get; set; }
        public DateTime? CreatedOn { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime? ModifiedOn { get; set; }
        public long? StatementRefNo { get; set; }
    }
}
