using System;

namespace Generated
{
    public class AgentsDiscountEntity
    {
        public int CustID { get; set; }
        public int ParentID { get; set; }
        public decimal DiscountOther { get; set; }
        public decimal DiscountGroupA { get; set; }
        public decimal DiscountGroupB { get; set; }
        public decimal DiscountGroupC { get; set; }
        public decimal Discount1x2 { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime ModifiedDate { get; set; }
    }
}
