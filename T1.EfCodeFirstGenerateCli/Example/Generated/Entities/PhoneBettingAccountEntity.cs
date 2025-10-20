using System;

namespace Generated
{
    public class PhoneBettingAccountEntity
    {
        public int CustID { get; set; }
        public required string UserName { get; set; }
        public bool Status { get; set; }
        public required string Remark { get; set; }
        public required string CreatedBy { get; set; }
        public DateTime? CreatedOn { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime? ModifiedOn { get; set; }
    }
}
