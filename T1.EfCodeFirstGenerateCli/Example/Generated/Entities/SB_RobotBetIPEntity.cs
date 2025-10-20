using System;

namespace Generated
{
    public class SB_RobotBetIPEntity
    {
        public int ID { get; set; }
        public required string IP { get; set; }
        public DateTime CreatedOn { get; set; }
        public DateTime ModifiedOn { get; set; }
        public required string ModifiedBy { get; set; }
        public bool IsNewIP { get; set; }
    }
}
