using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class customerEntityConfiguration : IEntityTypeConfiguration<customerEntity>
    {
        public void Configure(EntityTypeBuilder<customerEntity> builder)
        {
            builder.ToTable("customer");

            builder.HasKey(x => x.custid);

            builder.Property(x => x.custid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.firstname)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.lastname)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.address)
                .HasColumnType("nvarchar(80)")
                .HasMaxLength(80)
            ;

            builder.Property(x => x.postal)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.city)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.state)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.country)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.phone)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.mobilephone)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.fax)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.email)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.birthday)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.refcode1)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.refcode2)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.remark)
                .HasColumnType("nvarchar(150)")
                .HasMaxLength(150)
            ;

            builder.Property(x => x.creatdate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.creator)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.username)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.currency)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.creditcheck)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.userpwd)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.hitanswer)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.pwdtried)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.lastlogged)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.pwdhint)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.pwdexpiry)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.roleid)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.mrecommend)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.recommend)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.commaxpermatch)
                .HasColumnType("")
            ;

            builder.Property(x => x.closed)
                .HasColumnType("bit")
                .HasDefaultValue(false)
            ;

            builder.Property(x => x.dangercust)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.viewdetail)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.important)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.logincount)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.on_sessionid)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.site)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.mlive)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.alive)
                .HasColumnType("bit")
                .HasDefaultValue(true)
            ;

            builder.Property(x => x.lastloginIP)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.tstamp)
                .HasColumnType("timestamp")
            ;

            builder.Property(x => x.srecommend)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.slive)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(false)
            ;

            builder.Property(x => x.lastReadMsgID)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.sbo)
                .HasColumnType("tinyint(3,0)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.Subdomaingroup)
                .HasColumnType("varchar(10)")
                .HasMaxLength(10)
            ;

            builder.Property(x => x.isCash)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.status)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.title)
                .HasColumnType("nvarchar(10)")
                .HasMaxLength(10)
            ;

            builder.Property(x => x.gender)
                .HasColumnType("char(1)")
                .HasMaxLength(1)
            ;

            builder.Property(x => x.postalcode)
                .HasColumnType("nvarchar(20)")
                .HasMaxLength(20)
            ;

            builder.Property(x => x.timezone)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.SelfExclusionExpiredDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.period)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.stakelimit)
                .HasColumnType("")
            ;

            builder.Property(x => x.LimitExpiredDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.KYCExpiryDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.nickname)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.LoginName)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.LoginNameFlag)
                .HasColumnType("char(1)")
                .IsRequired()
                .HasMaxLength(1)
                .HasDefaultValue("N")
            ;

            builder.Property(x => x.Pin)
                .HasColumnType("char(4)")
                .HasMaxLength(4)
            ;

            builder.Property(x => x.Locality)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.LastLoginCountry)
                .HasColumnType("char(2)")
                .HasMaxLength(2)
            ;

            builder.Property(x => x.TCLastReadDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.LastLoggedOnUrl)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ExtraInfoID)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.PromotionEmail)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(true)
            ;

            builder.Property(x => x.DangerLevel)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.LockOutExpiryDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.CasinoMigrated)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(false)
            ;

            builder.Property(x => x.LastLoginProject)
                .HasColumnType("char(1)")
                .IsRequired()
                .HasMaxLength(1)
                .HasDefaultValue("i")
            ;

            builder.Property(x => x.MPDangerLevel)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.DisplayName)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ISOCurrency)
                .HasColumnType("char(3)")
                .HasMaxLength(3)
            ;

            builder.Property(x => x.CanChangeDisplayName)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.LastCasinoLoginUrl)
                .HasColumnType("nvarchar(100)")
                .HasMaxLength(100)
            ;

            builder.Property(x => x.CustomerGroupId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.DirectCustId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.SecurityMsg)
                .HasColumnType("nvarchar(30)")
                .HasMaxLength(30)
            ;

        }
    }
}
