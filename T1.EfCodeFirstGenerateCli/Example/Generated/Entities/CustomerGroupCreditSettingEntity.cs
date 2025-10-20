using System;

namespace Generated
{
    public class CustomerGroupCreditSettingEntity
    {
        public int CustomerGroupId { get; set; }
        public decimal GivenCredit { get; set; }
        public decimal SmaCreditLimit { get; set; }
        public decimal MaxCredit { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime ModifiedOn { get; set; }
    }
}
