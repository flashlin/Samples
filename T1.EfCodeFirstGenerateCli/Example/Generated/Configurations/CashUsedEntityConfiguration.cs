using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CashUsedEntityConfiguration : IEntityTypeConfiguration<CashUsedEntity>
    {
        public void Configure(EntityTypeBuilder<CashUsedEntity> builder)
        {
            builder.ToTable("CashUsed");

            builder.HasKey(x => x.custid);

            builder.Property(x => x.custid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.recommend)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.mrecommend)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.srecommend)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CashUsed)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.AgtCashUsed)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.MaCashUsed)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.SmaCashUsed)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.RBCashUsed)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.RBAgtCashUsed)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.RBMaCashUsed)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.RBSmaCashUsed)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.GMCashUsed)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.GMAgtCashUsed)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.GMMaCashUsed)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.GMSmaCashUsed)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.ServiceProvider)
                .HasColumnType("varchar(20)")
                .HasMaxLength(20)
            ;

            builder.Property(x => x.LastOrderOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.UserName)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.RToteCashUsed)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.RToteAgtCashUsed)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.RToteMaCashUsed)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.RToteSmaCashUsed)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.IsOutstanding)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.LCCashUsed)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.LCAgtCashUsed)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.LCMACashUsed)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.LCSMACashUsed)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.GMStakeLimit)
                .HasColumnType("decimal(23,6)")
            ;

            builder.Property(x => x.GMLimitExpiredDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.GMUsedStakeLimit)
                .HasColumnType("decimal(23,6)")
            ;

            builder.Property(x => x.tstamp)
                .HasColumnType("timestamp")
            ;

            builder.Property(x => x.SBStakeLimit)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.SBLimitExpiredDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.SBUsedStakeLimit)
                .HasColumnType("decimal(19,6)")
            ;

        }
    }
}
