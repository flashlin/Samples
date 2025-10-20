using System;

namespace Generated
{
    public class AgentFeatureEntity
    {
        public int CustID { get; set; }
        public int RoleID { get; set; }
        public required string OTPEmail { get; set; }
        public required string RecoveryEmail { get; set; }
        public required string Secret { get; set; }
        public bool? OTPNotRemindMe { get; set; }
    }
}
