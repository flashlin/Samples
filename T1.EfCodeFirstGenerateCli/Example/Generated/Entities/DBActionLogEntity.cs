using System;

namespace Generated
{
    public class DBActionLogEntity
    {
        public required string Action { get; set; }
        public DateTime? ActionTime { get; set; }
        public DateTime? ModifiedOn { get; set; }
        public required string ModifiedBy { get; set; }
    }
}
