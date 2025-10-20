using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class SuperSmaTagEntityConfiguration : IEntityTypeConfiguration<SuperSmaTagEntity>
    {
        public void Configure(EntityTypeBuilder<SuperSmaTagEntity> builder)
        {
            builder.ToTable("SuperSmaTag");

            builder.HasKey(x => x.CustomerId);

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.TagToCustomerId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.AuditGroupId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.SboContactCustomerId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

        }
    }
}
