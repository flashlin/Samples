using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class BatchLockEntityConfiguration : IEntityTypeConfiguration<BatchLockEntity>
    {
        public void Configure(EntityTypeBuilder<BatchLockEntity> builder)
        {
            builder.ToTable("BatchLock");

            builder.HasKey(x => x.BatchId);

            builder.Property(x => x.BatchId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.IsLock)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

        }
    }
}
