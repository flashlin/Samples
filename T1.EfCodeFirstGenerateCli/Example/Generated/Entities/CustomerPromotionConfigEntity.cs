using System;

namespace Generated
{
    public class CustomerPromotionConfigEntity
    {
        public int ID { get; set; }
        public int CustID { get; set; }
        public int PromotionType { get; set; }
        public decimal Target { get; set; }
        public DateTime? CreatedOn { get; set; }
        public required string CreatedBy { get; set; }
        public DateTime? ModifiedOn { get; set; }
        public required string ModifiedBy { get; set; }
    }
}
