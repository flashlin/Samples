using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class MoneyTransferLogDetailEntityConfiguration : IEntityTypeConfiguration<MoneyTransferLogDetailEntity>
    {
        public void Configure(EntityTypeBuilder<MoneyTransferLogDetailEntity> builder)
        {
            builder.ToTable("MoneyTransferLogDetail");

            builder.HasKey(x => x.MTDID);

            builder.Property(x => x.MTDID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.MTID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.TransferDetailType)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.FromID)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.FromAccountID)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ToID)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.ToAccountID)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Amount)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.MarketRate)
                .HasColumnType("decimal(12,8)")
            ;

            builder.Property(x => x.TransferDetailStatus)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.Description)
                .HasColumnType("nvarchar(200)")
                .HasMaxLength(200)
            ;

            builder.Property(x => x.Remark)
                .HasColumnType("varchar(200)")
                .HasMaxLength(200)
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.StatementRefNo)
                .HasColumnType("bigint(19,0)")
            ;

        }
    }
}
