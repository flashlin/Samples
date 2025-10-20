using System;

namespace Generated
{
    public class MpLoginNameAvailabilityEntity
    {
        public int id { get; set; }
        public required string server { get; set; }
        public required string UsrNameOrSessionID { get; set; }
        public required string LoginName { get; set; }
        public DateTime? GrantTime { get; set; }
        public bool Taken { get; set; }
        public required string ClientIP { get; set; }
    }
}
