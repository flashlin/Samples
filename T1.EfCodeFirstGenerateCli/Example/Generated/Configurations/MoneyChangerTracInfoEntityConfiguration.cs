using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class MoneyChangerTracInfoEntityConfiguration : IEntityTypeConfiguration<MoneyChangerTracInfoEntity>
    {
        public void Configure(EntityTypeBuilder<MoneyChangerTracInfoEntity> builder)
        {
            builder.ToTable("MoneyChangerTracInfo");

            builder.HasKey(x => x.Id);

            builder.Property(x => x.Id)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.TracDelayId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.InvoiceNumber)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Jurisdiction)
                .HasColumnType("varchar(10)")
                .IsRequired()
                .HasMaxLength(10)
            ;

            builder.Property(x => x.Status)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Batch)
                .HasColumnType("smallint(5,0)")
                .IsRequired()
            ;

            builder.Property(x => x.MoneyChangerSenderInfoId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.BankInfoId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.BankReceivedAmount)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.BankFee)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.SlipDetails)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Currency)
                .HasColumnType("nvarchar(5)")
                .IsRequired()
                .HasMaxLength(5)
            ;

            builder.Property(x => x.DealingDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.SlipDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.BankReceivedDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.UpdatedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.UpdatedBy)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Note)
                .HasColumnType("varchar(500)")
                .HasMaxLength(500)
            ;

            builder.Property(x => x.MoneyChangerTargetSenderInfoId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.PartnershipTargetSenderInfoId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.TargetSenderType)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
                .HasDefaultValue("None")
            ;

        }
    }
}
