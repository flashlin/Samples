using System;

namespace Generated
{
    public class KYCEmailNotifyEntity
    {
        public int custid { get; set; }
        public required string username { get; set; }
        public required string email { get; set; }
        public DateTime? KYCExpiryDate { get; set; }
        public bool? status { get; set; }
    }
}
