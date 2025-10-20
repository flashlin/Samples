using System;

namespace Generated
{
    public class LoginWhiteListEntity
    {
        public int LoginWhiteListId { get; set; }
        public int? CustId { get; set; }
        public required string AccountId { get; set; }
        public DateTime? FromDate { get; set; }
        public DateTime? ToDate { get; set; }
        public DateTime? CreatedOn { get; set; }
        public required string CreatedBy { get; set; }
        public required string Remark { get; set; }
    }
}
