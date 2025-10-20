using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class BankGroupMoneyChangerSenderInfoEntityConfiguration : IEntityTypeConfiguration<BankGroupMoneyChangerSenderInfoEntity>
    {
        public void Configure(EntityTypeBuilder<BankGroupMoneyChangerSenderInfoEntity> builder)
        {
            builder.ToTable("BankGroupMoneyChangerSenderInfo");

            builder.HasKey(x => x.Id);

            builder.Property(x => x.Id)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.MoneyChangerGroupId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.SenderName)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.UpdatedBy)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.UpdatedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.Status)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.ClosedDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.TargetGroupId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.IsRequireTargetSender)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(false)
            ;

            builder.Property(x => x.IsTargetSender)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(false)
            ;

        }
    }
}
