using System;

namespace Generated
{
    public class PlayerDiscountEntity
    {
        public int CustID { get; set; }
        public int AgtID { get; set; }
        public int MaID { get; set; }
        public int SmaID { get; set; }
        public decimal PlayerDiscount { get; set; }
        public decimal AgtDiscount { get; set; }
        public decimal MaDiscount { get; set; }
        public decimal SmaDiscount { get; set; }
        public decimal PlayerDiscount1x2 { get; set; }
        public decimal AgtDiscount1x2 { get; set; }
        public decimal MaDiscount1x2 { get; set; }
        public decimal SmaDiscount1x2 { get; set; }
        public decimal PlayerDiscountOther { get; set; }
        public decimal AgtDiscountOther { get; set; }
        public decimal MaDiscountOther { get; set; }
        public decimal SmaDiscountOther { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime ModifiedDate { get; set; }
        public required string Ugroup { get; set; }
    }
}
