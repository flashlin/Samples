using System;

namespace Generated
{
    public class MemberProdSettingEntity
    {
        public int CustID { get; set; }
        public int AgtID { get; set; }
        public int MaID { get; set; }
        public int SmaID { get; set; }
        public short HRMinDay { get; set; }
        public short HRMaxDay { get; set; }
        public bool HREnabled { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime ModifiedDate { get; set; }
        public short GMMinDay { get; set; }
        public short GMMaxDay { get; set; }
        public bool GMEnabled { get; set; }
        public bool CASEnabled { get; set; }
        public bool SBEnabled { get; set; }
        public bool RToteEnabled { get; set; }
        public bool RCEnabled { get; set; }
        public bool? RToteFollowEng { get; set; }
        public bool? LDEnabled { get; set; }
        public bool EFootballEnabled { get; set; }
    }
}
