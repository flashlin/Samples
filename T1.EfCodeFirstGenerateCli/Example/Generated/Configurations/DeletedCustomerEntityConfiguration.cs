using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class DeletedCustomerEntityConfiguration : IEntityTypeConfiguration<DeletedCustomerEntity>
    {
        public void Configure(EntityTypeBuilder<DeletedCustomerEntity> builder)
        {
            builder.ToTable("DeletedCustomer");

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

            builder.Property(x => x.maxbet)
                .HasColumnType("")
            ;

            builder.Property(x => x.minbet)
                .HasColumnType("")
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

            builder.Property(x => x.discount)
                .HasColumnType("")
            ;

            builder.Property(x => x.maxpermatch)
                .HasColumnType("")
            ;

            builder.Property(x => x.csmaxbet)
                .HasColumnType("")
            ;

            builder.Property(x => x.commaxbet)
                .HasColumnType("")
            ;

            builder.Property(x => x.csmaxpermatch)
                .HasColumnType("")
            ;

            builder.Property(x => x.commaxpermatch)
                .HasColumnType("")
            ;

            builder.Property(x => x.discountcom)
                .HasColumnType("")
            ;

            builder.Property(x => x.discountcs)
                .HasColumnType("")
            ;

            builder.Property(x => x.discount1x2)
                .HasColumnType("")
            ;

            builder.Property(x => x.positiontaking)
                .HasColumnType("")
            ;

            builder.Property(x => x.ugroup)
                .HasColumnType("varchar(5)")
                .HasMaxLength(5)
            ;

            builder.Property(x => x.playerdiscount)
                .HasColumnType("")
            ;

            builder.Property(x => x.playerdiscountcs)
                .HasColumnType("")
            ;

            builder.Property(x => x.playerdiscount1x2)
                .HasColumnType("")
            ;

            builder.Property(x => x.playerdiscountcom)
                .HasColumnType("")
            ;

            builder.Property(x => x.apositiontaking)
                .HasColumnType("")
            ;

            builder.Property(x => x.adiscount)
                .HasColumnType("")
            ;

            builder.Property(x => x.adiscountcs)
                .HasColumnType("")
            ;

            builder.Property(x => x.adiscount1x2)
                .HasColumnType("")
            ;

            builder.Property(x => x.adiscountcom)
                .HasColumnType("")
            ;

            builder.Property(x => x.mpositiontaking)
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

            builder.Property(x => x.groupa)
                .HasColumnType("")
            ;

            builder.Property(x => x.groupb)
                .HasColumnType("")
            ;

            builder.Property(x => x.groupc)
                .HasColumnType("")
            ;

            builder.Property(x => x.site)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.otpositiontaking)
                .HasColumnType("")
            ;

            builder.Property(x => x.ompositiontaking)
                .HasColumnType("")
            ;

            builder.Property(x => x.oapositiontaking)
                .HasColumnType("")
            ;

            builder.Property(x => x.mlive)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.alive)
                .HasColumnType("bit")
                .HasDefaultValue(true)
            ;

            builder.Property(x => x.lastloginIP)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.credit)
                .HasColumnType("")
                .IsRequired()
            ;

            builder.Property(x => x.tstamp)
                .HasColumnType("timestamp")
            ;

            builder.Property(x => x.mptlive1)
                .HasColumnType("decimal(3,2)")
            ;

            builder.Property(x => x.mptlive3)
                .HasColumnType("decimal(3,2)")
            ;

            builder.Property(x => x.aptlive1)
                .HasColumnType("decimal(3,2)")
            ;

            builder.Property(x => x.aptlive3)
                .HasColumnType("decimal(3,2)")
            ;

            builder.Property(x => x.sdiscount)
                .HasColumnType("")
                .IsRequired()
            ;

            builder.Property(x => x.sdiscountcs)
                .HasColumnType("")
                .IsRequired()
            ;

            builder.Property(x => x.spositiontaking)
                .HasColumnType("")
                .IsRequired()
            ;

            builder.Property(x => x.sptlive1)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.sptlive3)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
                .HasDefaultValue(0)
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

            builder.Property(x => x.ospositiontaking)
                .HasColumnType("")
                .IsRequired()
            ;

            builder.Property(x => x.lastReadMsgID)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.sbo)
                .HasColumnType("tinyint(3,0)")
            ;

            builder.Property(x => x.subdomaingroup)
                .HasColumnType("varchar(10)")
                .HasMaxLength(10)
            ;

            builder.Property(x => x.iscash)
                .HasColumnType("bit")
                .HasDefaultValue(false)
            ;

            builder.Property(x => x.DeletedDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.status)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.nickname)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.sdiscount1x2)
                .HasColumnType("")
                .IsRequired()
            ;

            builder.Property(x => x.selfexclusionexpireddate)
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

            builder.Property(x => x.kycexpirydate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.loginname)
                .HasColumnType("nvarchar(200)")
                .HasMaxLength(200)
            ;

            builder.Property(x => x.LoginNameFlag)
                .HasColumnType("char(1)")
                .HasMaxLength(1)
            ;

            builder.Property(x => x.DangerLevel)
                .HasColumnType("tinyint(3,0)")
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
            ;

            builder.Property(x => x.LockOutExpiryDate)
                .HasColumnType("datetime")
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
