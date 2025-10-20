using System;

namespace Generated
{
    public class CustomerRiskLimitEntity
    {
        public int CustID { get; set; }
        public int ParentID { get; set; }
        public byte RoleID { get; set; }
        public required string WinRisk { get; set; }
        public required string LostRisk { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime? ModifiedDate { get; set; }
    }
}
