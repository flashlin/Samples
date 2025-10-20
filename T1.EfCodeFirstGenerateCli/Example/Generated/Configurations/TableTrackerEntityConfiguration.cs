using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class TableTrackerEntityConfiguration : IEntityTypeConfiguration<TableTrackerEntity>
    {
        public void Configure(EntityTypeBuilder<TableTrackerEntity> builder)
        {
            builder.ToTable("TableTracker");

            builder.HasKey(x => x.rid);

            builder.Property(x => x.rid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.tablename)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.pk_id)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
            ;

            builder.Property(x => x.tstamp)
                .HasColumnType("timestamp")
                .IsRequired()
            ;

            builder.Property(x => x.createdon)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.LastModifiedOn)
                .HasColumnType("datetime")
            ;

        }
    }
}
