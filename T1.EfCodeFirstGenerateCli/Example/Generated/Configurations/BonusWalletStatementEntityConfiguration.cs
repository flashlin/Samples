using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class BonusWalletStatementEntityConfiguration : IEntityTypeConfiguration<BonusWalletStatementEntity>
    {
        public void Configure(EntityTypeBuilder<BonusWalletStatementEntity> builder)
        {
            builder.ToTable("BonusWalletStatement");

            builder.HasKey(x => x.TransId);

            builder.Property(x => x.TransId)
                .HasColumnType("bigint(19,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CustomerId)
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

            builder.Property(x => x.AgtId)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.MaId)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.SmaId)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.BonusWalletId)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.Remark)
                .HasColumnType("nvarchar(255)")
                .HasMaxLength(255)
            ;

            builder.Property(x => x.Description)
                .HasColumnType("nvarchar(300)")
                .HasMaxLength(300)
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

            builder.Property(x => x.StatementRefNo)
                .HasColumnType("bigint(19,0)")
            ;

            builder.Property(x => x.LastModifiedOn)
                .HasColumnType("datetime")
            ;

        }
    }
}
