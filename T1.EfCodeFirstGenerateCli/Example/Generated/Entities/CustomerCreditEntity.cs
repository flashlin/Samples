using System;

namespace Generated
{
    public class CustomerCreditEntity
    {
        public int CustId { get; set; }
        public int ParentId { get; set; }
        public byte AccountType { get; set; }
        public decimal PlayableLimit { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime? ModifiedOn { get; set; }
        public decimal PlayerMaxLimit { get; set; }
        public short TableLimit { get; set; }
        public decimal StakeLimit { get; set; }
        public DateTime LimitExpiredDate { get; set; }
        public decimal? DailyPlayerMaxLose { get; set; }
        public decimal? DailyPlayerMaxWin { get; set; }
        public bool? DailyResetEnabled { get; set; }
        public decimal? PlayableLimit1 { get; set; }
    }
}
