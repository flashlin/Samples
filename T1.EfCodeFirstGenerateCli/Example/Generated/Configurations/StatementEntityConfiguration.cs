using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class StatementEntityConfiguration : IEntityTypeConfiguration<StatementEntity>
    {
        public void Configure(EntityTypeBuilder<StatementEntity> builder)
        {
            builder.ToTable("Statement");

            builder.HasKey(x => x.Transid);

            builder.Property(x => x.Transid)
                .HasColumnType("bigint(19,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.Refno)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CustID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.UserName)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.TransDate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.WinLostDate)
                .HasColumnType("smalldatetime")
                .IsRequired()
            ;

            builder.Property(x => x.CheckTime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.AgtID)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.MaID)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.SmaID)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.TargetCustID)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.StatementType)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.Currency)
                .HasColumnType("varchar(10)")
                .HasMaxLength(10)
            ;

            builder.Property(x => x.CustomerStatus)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.ProductType)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.StatementStatus)
                .HasColumnType("int(10,0)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.IP)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
                .HasDefaultValue("")
            ;

            builder.Property(x => x.TransDesc)
                .HasColumnType("nvarchar(300)")
                .HasMaxLength(300)
            ;

            builder.Property(x => x.Remark)
                .HasColumnType("nvarchar(255)")
                .HasMaxLength(255)
            ;

            builder.Property(x => x.CreatorName)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CashIn)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.CashOut)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.CommIn)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.CommOut)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.DiscountIn)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.DiscountOut)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.ExternalRefno)
                .HasColumnType("nvarchar(30)")
                .HasMaxLength(30)
            ;

            builder.Property(x => x.ActualRate)
                .HasColumnType("decimal(12,8)")
            ;

            builder.Property(x => x.CasinoCashIn)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.CasinoCashOut)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.IsAdjusted)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.tstamp)
                .HasColumnType("timestamp")
            ;

            builder.Property(x => x.TargetParentGroupId)
                .HasColumnType("int(10,0)")
            ;

        }
    }
}
