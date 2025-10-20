using System;

namespace Generated
{
    public class SelfExclusionEmailNotifyEntity
    {
        public int custid { get; set; }
        public required string email { get; set; }
        public DateTime? SelfExclusionExpiredDate { get; set; }
        public bool? status { get; set; }
        public required string firstname { get; set; }
        public int? period { get; set; }
    }
}
