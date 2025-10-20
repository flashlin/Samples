using System;

namespace Generated
{
    public class customerEntity
    {
        public int custid { get; set; }
        public required string firstname { get; set; }
        public required string lastname { get; set; }
        public required string address { get; set; }
        public required string postal { get; set; }
        public required string city { get; set; }
        public required string state { get; set; }
        public required string country { get; set; }
        public required string phone { get; set; }
        public required string mobilephone { get; set; }
        public required string fax { get; set; }
        public required string email { get; set; }
        public DateTime? birthday { get; set; }
        public required string refcode1 { get; set; }
        public required string refcode2 { get; set; }
        public required string remark { get; set; }
        public DateTime? creatdate { get; set; }
        public int creator { get; set; }
        public required string username { get; set; }
        public int? currency { get; set; }
        public bool? creditcheck { get; set; }
        public required string userpwd { get; set; }
        public required string hitanswer { get; set; }
        public required string pwdtried { get; set; }
        public DateTime? lastlogged { get; set; }
        public required string pwdhint { get; set; }
        public DateTime? pwdexpiry { get; set; }
        public int? roleid { get; set; }
        public int? mrecommend { get; set; }
        public int? recommend { get; set; }
        public required string commaxpermatch { get; set; }
        public bool? closed { get; set; }
        public bool? dangercust { get; set; }
        public bool? viewdetail { get; set; }
        public bool? important { get; set; }
        public int? logincount { get; set; }
        public int? on_sessionid { get; set; }
        public required string site { get; set; }
        public bool? mlive { get; set; }
        public bool? alive { get; set; }
        public required string lastloginIP { get; set; }
        public DateTime? tstamp { get; set; }
        public int srecommend { get; set; }
        public bool slive { get; set; }
        public int? lastReadMsgID { get; set; }
        public byte? sbo { get; set; }
        public required string Subdomaingroup { get; set; }
        public bool? isCash { get; set; }
        public int status { get; set; }
        public required string title { get; set; }
        public required string gender { get; set; }
        public required string postalcode { get; set; }
        public byte? timezone { get; set; }
        public DateTime? SelfExclusionExpiredDate { get; set; }
        public int? period { get; set; }
        public required string stakelimit { get; set; }
        public DateTime? LimitExpiredDate { get; set; }
        public DateTime? KYCExpiryDate { get; set; }
        public required string nickname { get; set; }
        public required string LoginName { get; set; }
        public required string LoginNameFlag { get; set; }
        public required string Pin { get; set; }
        public required string Locality { get; set; }
        public required string LastLoginCountry { get; set; }
        public DateTime? TCLastReadDate { get; set; }
        public required string LastLoggedOnUrl { get; set; }
        public int? ExtraInfoID { get; set; }
        public bool PromotionEmail { get; set; }
        public byte? DangerLevel { get; set; }
        public DateTime? LockOutExpiryDate { get; set; }
        public bool CasinoMigrated { get; set; }
        public required string LastLoginProject { get; set; }
        public int MPDangerLevel { get; set; }
        public required string DisplayName { get; set; }
        public required string ISOCurrency { get; set; }
        public bool? CanChangeDisplayName { get; set; }
        public required string LastCasinoLoginUrl { get; set; }
        public int? CustomerGroupId { get; set; }
        public int? DirectCustId { get; set; }
        public required string SecurityMsg { get; set; }
    }
}
