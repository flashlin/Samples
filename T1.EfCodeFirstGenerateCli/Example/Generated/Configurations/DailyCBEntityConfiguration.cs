using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class DailyCBEntityConfiguration : IEntityTypeConfiguration<DailyCBEntity>
    {
        public void Configure(EntityTypeBuilder<DailyCBEntity> builder)
        {
            builder.ToTable("DailyCB");

            builder.HasKey(x => x.CBID);

            builder.Property(x => x.CBID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.ProductType)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.SubProductType)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.CBDate)
                .HasColumnType("date")
            ;

            builder.Property(x => x.FromID)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.FromAccount)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ToID)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.ToAccount)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Amount)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.TxnType)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.RefNo)
                .HasColumnType("bigint(19,0)")
            ;

            builder.Property(x => x.Description)
                .HasColumnType("varchar(100)")
                .HasMaxLength(100)
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

        }
    }
}
