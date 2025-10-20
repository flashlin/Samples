using System;

namespace Generated
{
    public class AgentMappingEntity
    {
        public required string ISOCurrency { get; set; }
        public required string AgentName { get; set; }
        public required string SubAgentName { get; set; }
        public bool Enable { get; set; }
        public required string ModifyBy { get; set; }
        public DateTime ModifyOn { get; set; }
    }
}
