using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CustomerVvipGroupEntityConfiguration : IEntityTypeConfiguration<CustomerVvipGroupEntity>
    {
        public void Configure(EntityTypeBuilder<CustomerVvipGroupEntity> builder)
        {
            builder.ToTable("CustomerVvipGroup");

            builder.HasKey(x => x.GroupId);

            builder.Property(x => x.GroupId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.GroupName)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
            ;

        }
    }
}
